int vnc_client_io_error(VncState *vs, int ret, int last_errno)

{

    if (ret == 0 || ret == -1) {

        if (ret == -1) {

            switch (last_errno) {

                case EINTR:

                case EAGAIN:

#ifdef _WIN32

                case WSAEWOULDBLOCK:

#endif

                    return 0;

                default:

                    break;

            }

        }



        VNC_DEBUG("Closing down client sock %d %d\n", ret, ret < 0 ? last_errno : 0);

        qemu_set_fd_handler2(vs->csock, NULL, NULL, NULL, NULL);

        closesocket(vs->csock);

        qemu_del_timer(vs->timer);

        qemu_free_timer(vs->timer);

        if (vs->input.buffer) qemu_free(vs->input.buffer);

        if (vs->output.buffer) qemu_free(vs->output.buffer);

#ifdef CONFIG_VNC_TLS

        vnc_tls_client_cleanup(vs);

#endif /* CONFIG_VNC_TLS */

#ifdef CONFIG_VNC_SASL

        vnc_sasl_client_cleanup(vs);

#endif /* CONFIG_VNC_SASL */

        audio_del(vs);



        VncState *p, *parent = NULL;

        for (p = vs->vd->clients; p != NULL; p = p->next) {

            if (p == vs) {

                if (parent)

                    parent->next = p->next;

                else

                    vs->vd->clients = p->next;

                break;

            }

            parent = p;

        }

        if (!vs->vd->clients)

            dcl->idle = 1;



        qemu_free(vs->server.ds->data);

        qemu_free(vs->server.ds);

        qemu_free(vs->guest.ds);

        qemu_free(vs);



        return 0;

    }

    return ret;

}
