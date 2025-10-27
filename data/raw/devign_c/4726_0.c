static void set_encodings(VncState *vs, int32_t *encodings, size_t n_encodings)

{

    int i;

    unsigned int enc = 0;



    vnc_zlib_init(vs);

    vs->features = 0;

    vs->vnc_encoding = -1;

    vs->tight_compression = 9;

    vs->tight_quality = 9;

    vs->absolute = -1;



    for (i = n_encodings - 1; i >= 0; i--) {

        enc = encodings[i];

        switch (enc) {

        case VNC_ENCODING_RAW:

            if (vs->vnc_encoding != -1) {

                vs->vnc_encoding = enc;

            }

            break;

        case VNC_ENCODING_COPYRECT:

            vs->features |= VNC_FEATURE_COPYRECT_MASK;

            break;

        case VNC_ENCODING_HEXTILE:

            vs->features |= VNC_FEATURE_HEXTILE_MASK;

            if (vs->vnc_encoding != -1) {

                vs->vnc_encoding = enc;

            }

            break;

        case VNC_ENCODING_ZLIB:

            vs->features |= VNC_FEATURE_ZLIB_MASK;

            if (vs->vnc_encoding != -1) {

                vs->vnc_encoding = enc;

            }

            break;

        case VNC_ENCODING_DESKTOPRESIZE:

            vs->features |= VNC_FEATURE_RESIZE_MASK;

            break;

        case VNC_ENCODING_POINTER_TYPE_CHANGE:

            vs->features |= VNC_FEATURE_POINTER_TYPE_CHANGE_MASK;

            break;

        case VNC_ENCODING_RICH_CURSOR:

            vs->features |= VNC_FEATURE_RICH_CURSOR_MASK;

            break;

        case VNC_ENCODING_EXT_KEY_EVENT:

            send_ext_key_event_ack(vs);

            break;

        case VNC_ENCODING_AUDIO:

            send_ext_audio_ack(vs);

            break;

        case VNC_ENCODING_WMVi:

            vs->features |= VNC_FEATURE_WMVI_MASK;

            break;

        case VNC_ENCODING_COMPRESSLEVEL0 ... VNC_ENCODING_COMPRESSLEVEL0 + 9:

            vs->tight_compression = (enc & 0x0F);

            break;

        case VNC_ENCODING_QUALITYLEVEL0 ... VNC_ENCODING_QUALITYLEVEL0 + 9:

            vs->tight_quality = (enc & 0x0F);

            break;

        default:

            VNC_DEBUG("Unknown encoding: %d (0x%.8x): %d\n", i, enc, enc);

            break;

        }

    }

    check_pointer_type_change(&vs->mouse_mode_notifier);

}
