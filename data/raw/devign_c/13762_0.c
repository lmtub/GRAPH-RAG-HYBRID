void vnc_zlib_clear(VncState *vs)

{

    if (vs->zlib_stream.opaque) {

        deflateEnd(&vs->zlib_stream);

    }

    buffer_free(&vs->zlib);

}
