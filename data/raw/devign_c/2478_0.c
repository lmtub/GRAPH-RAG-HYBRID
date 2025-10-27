static int protocol_client_auth_vnc(VncState *vs, uint8_t *data, size_t len)

{

    unsigned char response[VNC_AUTH_CHALLENGE_SIZE];

    int i, j, pwlen;

    unsigned char key[8];

    time_t now = time(NULL);



    if (!vs->vd->password || !vs->vd->password[0]) {

        VNC_DEBUG("No password configured on server");

        goto reject;

    }

    if (vs->vd->expires < now) {

        VNC_DEBUG("Password is expired");

        goto reject;

    }



    memcpy(response, vs->challenge, VNC_AUTH_CHALLENGE_SIZE);



    /* Calculate the expected challenge response */

    pwlen = strlen(vs->vd->password);

    for (i=0; i<sizeof(key); i++)

        key[i] = i<pwlen ? vs->vd->password[i] : 0;

    deskey(key, EN0);

    for (j = 0; j < VNC_AUTH_CHALLENGE_SIZE; j += 8)

        des(response+j, response+j);



    /* Compare expected vs actual challenge response */

    if (memcmp(response, data, VNC_AUTH_CHALLENGE_SIZE) != 0) {

        VNC_DEBUG("Client challenge reponse did not match\n");

        goto reject;

    } else {

        VNC_DEBUG("Accepting VNC challenge response\n");

        vnc_write_u32(vs, 0); /* Accept auth */

        vnc_flush(vs);



        start_client_init(vs);

    }

    return 0;



reject:

    vnc_write_u32(vs, 1); /* Reject auth */

    if (vs->minor >= 8) {

        static const char err[] = "Authentication failed";

        vnc_write_u32(vs, sizeof(err));

        vnc_write(vs, err, sizeof(err));

    }

    vnc_flush(vs);

    vnc_client_error(vs);

    return 0;

}
