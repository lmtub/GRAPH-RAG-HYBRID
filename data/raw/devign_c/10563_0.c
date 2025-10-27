char *ff_get_ref_perms_string(char *buf, size_t buf_size, int perms)

{

    snprintf(buf, buf_size, "%s%s%s%s%s",

             perms & AV_PERM_READ      ? "r" : "",

             perms & AV_PERM_WRITE     ? "w" : "",

             perms & AV_PERM_PRESERVE  ? "p" : "",

             perms & AV_PERM_REUSE     ? "u" : "",

             perms & AV_PERM_REUSE2    ? "U" : "");

    return buf;

}
