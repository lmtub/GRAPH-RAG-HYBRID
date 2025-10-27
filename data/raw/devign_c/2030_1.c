void error_setg_file_open(Error **errp, int os_errno, const char *filename)

{

    error_setg_errno(errp, os_errno, "Could not open '%s'", filename);

}
