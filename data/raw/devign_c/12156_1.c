static void blkreplay_bh_cb(void *opaque)

{

    Request *req = opaque;

    qemu_coroutine_enter(req->co, NULL);

    qemu_bh_delete(req->bh);

    g_free(req);

}
