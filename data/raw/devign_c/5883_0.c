static int v9fs_do_chown(V9fsState *s, V9fsString *path, uid_t uid, gid_t gid)

{

    return s->ops->chown(&s->ctx, path->data, uid, gid);

}
