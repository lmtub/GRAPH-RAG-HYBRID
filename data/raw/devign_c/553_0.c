void vnc_disconnect_finish(VncState *vs)

{

    int i;



    vnc_jobs_join(vs); /* Wait encoding jobs */



    vnc_lock_output(vs);

    vnc_qmp_event(vs, QAPI_EVENT_VNC_DISCONNECTED);



    buffer_free(&vs->input);

    buffer_free(&vs->output);

#ifdef CONFIG_VNC_WS

    buffer_free(&vs->ws_input);

    buffer_free(&vs->ws_output);

#endif /* CONFIG_VNC_WS */



    qapi_free_VncClientInfo(vs->info);



    vnc_zlib_clear(vs);

    vnc_tight_clear(vs);

    vnc_zrle_clear(vs);



#ifdef CONFIG_VNC_TLS

    vnc_tls_client_cleanup(vs);

#endif /* CONFIG_VNC_TLS */

#ifdef CONFIG_VNC_SASL

    vnc_sasl_client_cleanup(vs);

#endif /* CONFIG_VNC_SASL */

    audio_del(vs);

    vnc_release_modifiers(vs);



    if (vs->initialized) {

        QTAILQ_REMOVE(&vs->vd->clients, vs, next);

        qemu_remove_mouse_mode_change_notifier(&vs->mouse_mode_notifier);

    }



    if (vs->vd->lock_key_sync)

        qemu_remove_led_event_handler(vs->led);

    vnc_unlock_output(vs);



    qemu_mutex_destroy(&vs->output_mutex);

    if (vs->bh != NULL) {

        qemu_bh_delete(vs->bh);

    }

    buffer_free(&vs->jobs_buffer);



    for (i = 0; i < VNC_STAT_ROWS; ++i) {

        g_free(vs->lossy_rect[i]);

    }

    g_free(vs->lossy_rect);

    g_free(vs);

}
