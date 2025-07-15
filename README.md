# ShearWallDesignToools
This repository contains the master code that is being used by the excel spreadsheet

## Logging

The Python modules use the standard `logging` library. The log level can be
configured in two ways:

1. Set the environment variable `LOG_LEVEL` to a valid logging level such as
   `DEBUG`, `INFO`, etc.
2. Provide the desired level in cell `B3` of the Excel workbook. If both are
   omitted, the log level defaults to `INFO`.
