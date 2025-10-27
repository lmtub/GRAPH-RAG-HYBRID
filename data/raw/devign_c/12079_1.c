static void file_completion(const char *input)

{

    DIR *ffs;

    struct dirent *d;

    char path[1024];

    char file[1024], file_prefix[1024];

    int input_path_len;

    const char *p;



    p = strrchr(input, '/');

    if (!p) {

        input_path_len = 0;

        pstrcpy(file_prefix, sizeof(file_prefix), input);

        pstrcpy(path, sizeof(path), ".");

    } else {

        input_path_len = p - input + 1;

        memcpy(path, input, input_path_len);

        if (input_path_len > sizeof(path) - 1)

            input_path_len = sizeof(path) - 1;

        path[input_path_len] = '\0';

        pstrcpy(file_prefix, sizeof(file_prefix), p + 1);

    }

#ifdef DEBUG_COMPLETION

    monitor_printf(cur_mon, "input='%s' path='%s' prefix='%s'\n",

                   input, path, file_prefix);

#endif

    ffs = opendir(path);

    if (!ffs)

        return;

    for(;;) {

        struct stat sb;

        d = readdir(ffs);

        if (!d)

            break;



        if (strcmp(d->d_name, ".") == 0 || strcmp(d->d_name, "..") == 0) {

            continue;

        }



        if (strstart(d->d_name, file_prefix, NULL)) {

            memcpy(file, input, input_path_len);

            if (input_path_len < sizeof(file))

                pstrcpy(file + input_path_len, sizeof(file) - input_path_len,

                        d->d_name);

            /* stat the file to find out if it's a directory.

             * In that case add a slash to speed up typing long paths

             */

            stat(file, &sb);

            if(S_ISDIR(sb.st_mode))

                pstrcat(file, sizeof(file), "/");

            readline_add_completion(cur_mon->rs, file);

        }

    }

    closedir(ffs);

}
