static int file_read_dir(URLContext *h, AVIODirEntry **next)

{

#if HAVE_DIRENT_H

    FileContext *c = h->priv_data;

    struct dirent *dir;

    char *fullpath = NULL;



    *next = ff_alloc_dir_entry();

    if (!*next)

        return AVERROR(ENOMEM);

    do {

        errno = 0;

        dir = readdir(c->dir);

        if (!dir) {

            av_freep(next);

            return AVERROR(errno);

        }

    } while (!strcmp(dir->d_name, ".") || !strcmp(dir->d_name, ".."));



    fullpath = av_append_path_component(h->filename, dir->d_name);

    if (fullpath) {

        struct stat st;

        if (!lstat(fullpath, &st)) {

            if (S_ISDIR(st.st_mode))

                (*next)->type = AVIO_ENTRY_DIRECTORY;

            else if (S_ISFIFO(st.st_mode))

                (*next)->type = AVIO_ENTRY_NAMED_PIPE;

            else if (S_ISCHR(st.st_mode))

                (*next)->type = AVIO_ENTRY_CHARACTER_DEVICE;

            else if (S_ISBLK(st.st_mode))

                (*next)->type = AVIO_ENTRY_BLOCK_DEVICE;

            else if (S_ISLNK(st.st_mode))

                (*next)->type = AVIO_ENTRY_SYMBOLIC_LINK;

            else if (S_ISSOCK(st.st_mode))

                (*next)->type = AVIO_ENTRY_SOCKET;

            else if (S_ISREG(st.st_mode))

                (*next)->type = AVIO_ENTRY_FILE;

            else

                (*next)->type = AVIO_ENTRY_UNKNOWN;



            (*next)->group_id = st.st_gid;

            (*next)->user_id = st.st_uid;

            (*next)->size = st.st_size;

            (*next)->filemode = st.st_mode & 0777;

            (*next)->modification_timestamp = INT64_C(1000000) * st.st_mtime;

            (*next)->access_timestamp =  INT64_C(1000000) * st.st_atime;

            (*next)->status_change_timestamp = INT64_C(1000000) * st.st_ctime;

        }

        av_free(fullpath);

    }



    (*next)->name = av_strdup(dir->d_name);

    return 0;

#else

    return AVERROR(ENOSYS);

#endif /* HAVE_DIRENT_H */

}
