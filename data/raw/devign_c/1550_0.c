static int v9fs_do_symlink(V9fsState *s, V9fsString *oldpath,

                            V9fsString *newpath)

{

    return s->ops->symlink(&s->ctx, oldpath->data, newpath->data);

}
