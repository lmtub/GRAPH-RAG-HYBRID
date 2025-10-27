static int v9fs_do_lstat(V9fsState *s, V9fsString *path, struct stat *stbuf)

{

    return s->ops->lstat(&s->ctx, path->data, stbuf);

}
