# By ZhuMengjing
"""Library to run MMseqs2 from Python."""

import os
import subprocess
from typing import Any, Mapping, Optional, Sequence

from absl import logging

from helixfold.data import parsers
from helixfold.data.tools import utils


class MMseqs2:
    """Python wrapper of the MMseqs2 binary (GPU-only version)."""

    def __init__(
        self,
        *,
        binary_path: str,
        database_path: str,
        e_value: float = 0.0001,
        max_sequences: int = 10000,
        sensitivity: float = 7.5,
        gpu_devices: Optional[Sequence[str]] = None):
        """Initializes the Python MMseqs2 wrapper (GPU-only).

        Args:
            binary_path: Path to the MMseqs2 binary.
            database_path: Path to the sequence database.
            e_value: E-value cutoff.
            max_sequences: Maximum number of sequences to return.
            sensitivity: Search sensitivity (from -s parameter).
            gpu_devices: Optional list of GPU device identifiers. If None, the
                value is inferred from the current environment variables.
        """
        self.binary_path = binary_path
        self.database_path = database_path

        if not os.path.exists(self.database_path):
            logging.error("Could not find MMseqs2 database %s", database_path)
            raise ValueError(f"Could not find MMseqs2 database {database_path}")

        self.e_value = e_value
        self.max_sequences = max_sequences
        self.sensitivity = sensitivity
        if gpu_devices is None:
            gpu_devices = self._resolve_gpu_devices_from_env()

        # Normalise into an immutable container so we can safely reuse the
        # resolved devices for every subprocess invocation.
        self.gpu_devices = tuple(gpu_devices)


    def _resolve_gpu_devices_from_env(self) -> Sequence[str]:
        """Determine visible GPU devices from environment variables."""

        # Prefer an explicit override so that callers can fully control
        # the GPU visibility without modifying code.
        explicit = os.environ.get("MMSEQS_GPU_DEVICES")
        if explicit:
            devices = [d.strip() for d in explicit.split(",") if d.strip()]
            if devices:
                return devices

        # Fall back to the CUDA/NVIDIA visibility settings inherited from the
        # launcher (e.g. docker run --gpus or CUDA_VISIBLE_DEVICES).
        for env_name in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"):
            value = os.environ.get(env_name)
            if value and value.lower() != "all":
                devices = [d.strip() for d in value.split(",") if d.strip()]
                if devices:
                    return devices

        # If nothing is specified, default to the first GPU (index 0).
        return ("0",)

  
    def _gpu_flag_argument(self) -> str:
        """Return the number of GPUs to pass to the ``--gpu`` flag."""

        if not self.gpu_devices:
            return "1"

        # ``--gpu`` expects a count, not an explicit device list. We still
        # honour the resolved devices through ``CUDA_VISIBLE_DEVICES`` but make
        # sure we request the correct number of GPUs here.
        active_devices = [device for device in self.gpu_devices if device]
        return str(len(active_devices)) if active_devices else "1"

    def _run_command(self, cmd: Sequence[str], *, use_gpu_env: bool = True) -> None:
        """Runs command with proper environment and logging."""
        env = os.environ.copy()

        if use_gpu_env and self.gpu_devices:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(self.gpu_devices)
        elif not use_gpu_env:
            # Ensure GPU visibility constraints are removed for CPU fallbacks.
            env.pop("CUDA_VISIBLE_DEVICES", None)
            env.pop("NVIDIA_VISIBLE_DEVICES", None)

        logging.info('Running command: %s', ' '.join(cmd))
        logging.debug(
            "Using CUDA_VISIBLE_DEVICES=%s", env.get("CUDA_VISIBLE_DEVICES", "<unset>")
        )
        try:
            subprocess.run(
                cmd, env=env, capture_output=True, check=True, text=True
            )
        except subprocess.CalledProcessError as exc:
            if exc.stdout:
                logging.error("stdout from failed MMseqs2 command:\n%s", exc.stdout)
            if exc.stderr:
                logging.error("stderr from failed MMseqs2 command:\n%s", exc.stderr)
            raise

    def _should_retry_on_cpu(self, exc: subprocess.CalledProcessError) -> bool:
        """Return True if the failure should trigger a CPU retry."""

        failure_output = ""
        if exc.stdout:
            failure_output += exc.stdout
        if exc.stderr:
            failure_output += exc.stderr

        oom_signatures = (
            "CUDA error: out of memory",
            "Ungapped prefilter died",
        )
        return any(signature in failure_output for signature in oom_signatures)

    @staticmethod
    def _strip_gpu_flags(cmd: Sequence[str]) -> Sequence[str]:
        """Remove GPU-related flags from the provided command sequence."""

        stripped = []
        skip_next = False
        for token in cmd:
            if skip_next:
                skip_next = False
                continue
            if token == "--gpu":
                skip_next = True
                continue
            stripped.append(token)
        return stripped
        # process = subprocess.run(
        #     cmd, env=env, capture_output=True, check=True, text=True
        # )
        
        # if process.stdout:
        #     logging.info("stdout:\n%s\n", process.stdout)
        # if process.stderr:
        #     logging.info("stderr:\n%s\n", process.stderr)

    def _query_chunk(
        self,
        input_fasta_path: str,
        database_path: str,
        max_sequences: Optional[int] = None,
    ) -> Mapping[str, Any]:
        """Queries a database chunk using MMseqs2 GPU."""
        with utils.tmpdir_manager(base_dir="/tmp") as query_tmp_dir:
            result_sto = os.path.join(query_tmp_dir, f"result.sto")
            
            # Create query DB
            query_db = os.path.join(query_tmp_dir, "queryDB")
            cmd = [self.binary_path, "createdb", input_fasta_path, query_db]
            self._run_command(cmd)

            # Run GPU search
            db_gpu = os.path.join(database_path, 'targetDB_gpu')
            result_db = os.path.join(query_tmp_dir, "resultDB")

            cmd = [
                self.binary_path,
                "search",
                query_db,
                db_gpu,
                result_db,
                query_tmp_dir,
                "--gpu",
                self._gpu_flag_argument(),
                #"--gpu-server", "1",
                "--db-load-mode",
                "2",
            ]
            try:
                self._run_command(cmd)
            except subprocess.CalledProcessError as exc:
                if not self._should_retry_on_cpu(exc):
                    raise
                logging.warning(
                    "MMseqs2 GPU search failed due to GPU resource limits; retrying on CPU"
                )
                cpu_cmd = self._strip_gpu_flags(cmd)
                self._run_command(cpu_cmd, use_gpu_env=False)

            # Convert to sto format
            cmd = [
                self.binary_path,
                "result2msa",
                query_db,
                db_gpu,
                result_db,
                result_sto,
                "--msa-format-mode", "4",
                "--db-load-mode", "2",
            ]
            self._run_command(cmd)

            # Read and process results
            with open(result_sto) as f:
                sto = f.read()

            if max_sequences is not None:
                sto = parsers.truncate_stockholm_msa(result_sto, max_sequences)

            return {
                "sto": sto,
                "tbl": "",  # MMseqs2 doesn't produce tblout by default
                "stderr": "",  # stderr is already logged
                "n_iter": 1,  # MMseqs2 doesn't use iterations like Jackhmmer
                "e_value": self.e_value,
            }

    def query(
        self,
        input_fasta_path: str,
        max_sequences: Optional[int] = None,
    ) -> Sequence[Mapping[str, Any]]:
        """Queries the database using MMseqs2 GPU."""

        single_chunk_result = self._query_chunk(
            input_fasta_path, self.database_path, max_sequences
        )
        return [single_chunk_result]
