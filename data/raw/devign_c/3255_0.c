static int v9fs_do_mksock(V9fsState *s, V9fsString *path)

{

    return s->ops->mksock(&s->ctx, path->data);

}
