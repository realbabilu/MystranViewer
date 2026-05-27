"""
OU1 Reader Module
=================

This module reads MYSTRAN OUTPUT4 binary matrix files (.OU1).

The OU1 file is written as Fortran unformatted sequential records.
Each matrix is stored as:

    HEADER RECORD (24 bytes total)
        int32 nrows
        int32 ncols
        int32 form
        int32 type
        char[8] name

    FOLLOWED BY ncols COLUMN RECORDS
        int32 column_index (1-based)
        int32 first_row
        int32 nvalues
        float64[nvalues] column_data

Notes
-----
- Column data is stored in column-major fashion.
- DOF numbering in MYSTRAN is 1-based.
- Matrix values are stored as float64.
"""

from typing import Dict
import numpy as np
from scipy.io import FortranFile


def read_ou1(filename: str) -> Dict[str, np.ndarray]:
    """
    Read MYSTRAN OUTPUT4 (.OU1) binary matrix file.

    Parameters
    ----------
    filename : str
        Path to the OU1 file.

    Returns
    -------
    matrices : dict[str, np.ndarray]
        Dictionary mapping matrix name -> dense NumPy array.

        Example:
            matrices["KAA"]  -> stiffness matrix
            matrices["MAA"]  -> mass matrix

    Notes
    -----
    - Only full (dense) matrices are supported.
    - End-of-matrix marker records are automatically skipped.
    - Returned matrices use zero-based indexing.
    """

    matrices: Dict[str, np.ndarray] = {}

    with FortranFile(filename, 'r') as f:

        while True:
            try:
                rec = f.read_record(np.uint8)
            except Exception:
                # Proper EOF reached
                break

            # -------------------------------------------------
            # Validate header record size
            # -------------------------------------------------
            if len(rec) != 24:
                continue

            header = np.frombuffer(rec[:16], dtype=np.int32)

            if header.size < 2:
                continue

            nrows = int(header[0])
            ncols = int(header[1])

            name = rec[16:24].tobytes().decode(errors='ignore').strip()

            # -------------------------------------------------
            # Skip invalid or marker records
            # -------------------------------------------------
            if nrows <= 0 or ncols <= 1 or name == "":
                continue

            print(f"Reading matrix '{name}' ({nrows} x {ncols})")

            mat = np.zeros((nrows, ncols), dtype=np.float64)

            # -------------------------------------------------
            # Read column records
            # -------------------------------------------------
            for _ in range(ncols):

                rec = f.read_record(np.uint8)

                if len(rec) < 12:
                    raise ValueError(
                        f"Invalid column record encountered in matrix '{name}'."
                    )

                col_header = np.frombuffer(rec[:12], dtype=np.int32)

                if col_header.size < 1:
                    raise ValueError(
                        f"Corrupted column header in matrix '{name}'."
                    )

                col_index = int(col_header[0]) - 1  # convert 1-based → 0-based

                data = np.frombuffer(rec[12:], dtype=np.float64)

                if data.size != nrows:
                    raise ValueError(
                        f"Column size mismatch in matrix '{name}'. "
                        f"Expected {nrows}, got {data.size}."
                    )

                mat[:, col_index] = data

            matrices[name] = mat

    return matrices