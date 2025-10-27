static int v9fs_do_open2(V9fsState *s, V9fsString *path, int flags, mode_t mode)

{

    return s->ops->open2(&s->ctx, path->data, flags, mode);

}
