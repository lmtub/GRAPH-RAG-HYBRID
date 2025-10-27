static int v9fs_do_setuid(V9fsState *s, uid_t uid)

{

    return s->ops->setuid(&s->ctx, uid);

}
