void vnc_tight_clear(VncState *vs)

{

    int i;

    for (i=0; i<ARRAY_SIZE(vs->tight.stream); i++) {

        if (vs->tight.stream[i].opaque) {

            deflateEnd(&vs->tight.stream[i]);

        }

    }



    buffer_free(&vs->tight.tight);

    buffer_free(&vs->tight.zlib);

    buffer_free(&vs->tight.gradient);

#ifdef CONFIG_VNC_JPEG

    buffer_free(&vs->tight.jpeg);





}