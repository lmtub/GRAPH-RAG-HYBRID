char *qemu_find_file(int type, const char *name)

{

    int len;

    const char *subdir;

    char *buf;



    /* Try the name as a straight path first */

    if (access(name, R_OK) == 0) {

        return g_strdup(name);

    }

    switch (type) {

    case QEMU_FILE_TYPE_BIOS:

        subdir = "";

        break;

    case QEMU_FILE_TYPE_KEYMAP:

        subdir = "keymaps/";

        break;

    default:

        abort();

    }

    len = strlen(data_dir) + strlen(name) + strlen(subdir) + 2;

    buf = g_malloc0(len);

    snprintf(buf, len, "%s/%s%s", data_dir, subdir, name);

    if (access(buf, R_OK)) {

        g_free(buf);

        return NULL;

    }

    return buf;

}
