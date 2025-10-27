static void vnc_disconnect_finish(VncState *vs)

{

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



    vnc_remove_timer(vs->vd);

    qemu_free(vs);

}
