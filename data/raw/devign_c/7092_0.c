static int v9fs_do_mkdir(V9fsState *s, V9fsString *path, mode_t mode)

{

    return s->ops->mkdir(&s->ctx, path->data, mode);

}
