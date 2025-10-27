static int start_auth_vencrypt_subauth(VncState *vs)

{

    switch (vs->vd->subauth) {

    case VNC_AUTH_VENCRYPT_TLSNONE:

    case VNC_AUTH_VENCRYPT_X509NONE:

       VNC_DEBUG("Accept TLS auth none\n");

       vnc_write_u32(vs, 0); /* Accept auth completion */

       vnc_read_when(vs, protocol_client_init, 1);

       break;



    case VNC_AUTH_VENCRYPT_TLSVNC:

    case VNC_AUTH_VENCRYPT_X509VNC:

       VNC_DEBUG("Start TLS auth VNC\n");

       return start_auth_vnc(vs);



    default: /* Should not be possible, but just in case */

       VNC_DEBUG("Reject auth %d\n", vs->vd->auth);

       vnc_write_u8(vs, 1);

       if (vs->minor >= 8) {

           static const char err[] = "Unsupported authentication type";

           vnc_write_u32(vs, sizeof(err));

           vnc_write(vs, err, sizeof(err));

       }

       vnc_client_error(vs);

    }



    return 0;

}
