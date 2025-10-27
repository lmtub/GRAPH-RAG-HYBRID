static int file_close_dir(URLContext *h)

{

#if HAVE_DIRENT_H

    FileContext *c = h->priv_data;

    closedir(c->dir);

    return 0;

#else

    return AVERROR(ENOSYS);

#endif /* HAVE_DIRENT_H */

}
