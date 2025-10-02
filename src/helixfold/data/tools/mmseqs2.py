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
        gpu_devices: Sequence[str] = ("4", "5", "6")):
        """Initializes the Python MMseqs2 wrapper (GPU-only).

        Args:
            binary_path: Path to the MMseqs2 binary.
            database_path: Path to the sequence database.
            e_value: E-value cutoff.
            max_sequences: Maximum number of sequences to return.
            sensitivity: Search sensitivity (from -s parameter).
            gpu_devices: List of GPU devices to use (e.g. ["0", "1"]).
        """
        self.binary_path = binary_path
        self.database_path = database_path

        if not os.path.exists(self.database_path):
            logging.error("Could not find MMseqs2 database %s", database_path)
            raise ValueError(f"Could not find MMseqs2 database {database_path}")

        self.e_value = e_value
        self.max_sequences = max_sequences
        self.sensitivity = sensitivity
        self.gpu_devices = gpu_devices

  
    def _run_command(self, cmd: Sequence[str]) -> None:
        """Runs command with proper environment and logging."""
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(self.gpu_devices)
        
        logging.info('Running command: %s', ' '.join(cmd))
        subprocess.run(
            cmd, env=env, capture_output=True, check=True, text=True
        )
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
                "--gpu", "1",
                #"--gpu-server", "1",
                "--db-load-mode", "2",
            ]
            self._run_command(cmd)

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