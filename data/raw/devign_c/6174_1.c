static void qmp_query_auth(VncDisplay *vd, VncInfo2 *info)

{

    switch (vd->auth) {

    case VNC_AUTH_VNC:

        info->auth = VNC_PRIMARY_AUTH_VNC;

        break;

    case VNC_AUTH_RA2:

        info->auth = VNC_PRIMARY_AUTH_RA2;

        break;

    case VNC_AUTH_RA2NE:

        info->auth = VNC_PRIMARY_AUTH_RA2NE;

        break;

    case VNC_AUTH_TIGHT:

        info->auth = VNC_PRIMARY_AUTH_TIGHT;

        break;

    case VNC_AUTH_ULTRA:

        info->auth = VNC_PRIMARY_AUTH_ULTRA;

        break;

    case VNC_AUTH_TLS:

        info->auth = VNC_PRIMARY_AUTH_TLS;

        break;

    case VNC_AUTH_VENCRYPT:

        info->auth = VNC_PRIMARY_AUTH_VENCRYPT;

#ifdef CONFIG_VNC_TLS

        info->has_vencrypt = true;

        switch (vd->subauth) {

        case VNC_AUTH_VENCRYPT_PLAIN:

            info->vencrypt = VNC_VENCRYPT_SUB_AUTH_PLAIN;

            break;

        case VNC_AUTH_VENCRYPT_TLSNONE:

            info->vencrypt = VNC_VENCRYPT_SUB_AUTH_TLS_NONE;

            break;

        case VNC_AUTH_VENCRYPT_TLSVNC:

            info->vencrypt = VNC_VENCRYPT_SUB_AUTH_TLS_VNC;

            break;

        case VNC_AUTH_VENCRYPT_TLSPLAIN:

            info->vencrypt = VNC_VENCRYPT_SUB_AUTH_TLS_PLAIN;

            break;

        case VNC_AUTH_VENCRYPT_X509NONE:

            info->vencrypt = VNC_VENCRYPT_SUB_AUTH_X509_NONE;

            break;

        case VNC_AUTH_VENCRYPT_X509VNC:

            info->vencrypt = VNC_VENCRYPT_SUB_AUTH_X509_VNC;

            break;

        case VNC_AUTH_VENCRYPT_X509PLAIN:

            info->vencrypt = VNC_VENCRYPT_SUB_AUTH_X509_PLAIN;

            break;

        case VNC_AUTH_VENCRYPT_TLSSASL:

            info->vencrypt = VNC_VENCRYPT_SUB_AUTH_TLS_SASL;

            break;

        case VNC_AUTH_VENCRYPT_X509SASL:

            info->vencrypt = VNC_VENCRYPT_SUB_AUTH_X509_SASL;

            break;

        default:

            info->has_vencrypt = false;

            break;

        }

#endif

        break;

    case VNC_AUTH_SASL:

        info->auth = VNC_PRIMARY_AUTH_SASL;

        break;

    case VNC_AUTH_NONE:

    default:

        info->auth = VNC_PRIMARY_AUTH_NONE;

        break;

    }

}
