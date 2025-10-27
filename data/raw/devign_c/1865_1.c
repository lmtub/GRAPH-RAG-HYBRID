static void qemu_add_data_dir(const char *path)

{

    int i;



    if (path == NULL) {

        return;

    }

    if (data_dir_idx == ARRAY_SIZE(data_dir)) {

        return;

    }

    for (i = 0; i < data_dir_idx; i++) {

        if (strcmp(data_dir[i], path) == 0) {

            return; /* duplicate */

        }

    }

    data_dir[data_dir_idx++] = path;

}
