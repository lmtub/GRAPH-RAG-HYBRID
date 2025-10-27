static int rtmp_handshake(URLContext *s, RTMPContext *rt)
{
    AVLFG rnd;
    uint8_t tosend    [RTMP_HANDSHAKE_PACKET_SIZE+1] = {
        3,                // unencrypted data
        0, 0, 0, 0,       // client uptime
        RTMP_CLIENT_VER1,
        RTMP_CLIENT_VER2,
        RTMP_CLIENT_VER3,
        RTMP_CLIENT_VER4,
    };
    uint8_t clientdata[RTMP_HANDSHAKE_PACKET_SIZE];
    uint8_t serverdata[RTMP_HANDSHAKE_PACKET_SIZE+1];
    int i;
    int server_pos, client_pos;
    uint8_t digest[32], signature[32];
    int ret, type = 0;
    av_log(s, AV_LOG_DEBUG, "Handshaking...\n");
    av_lfg_init(&rnd, 0xDEADC0DE);
    // generate handshake packet - 1536 bytes of pseudorandom data
    for (i = 9; i <= RTMP_HANDSHAKE_PACKET_SIZE; i++)
        tosend[i] = av_lfg_get(&rnd) >> 24;
    if (rt->encrypted && CONFIG_FFRTMPCRYPT_PROTOCOL) {
        /* When the client wants to use RTMPE, we have to change the command
         * byte to 0x06 which means to use encrypted data and we have to set
         * the flash version to at least 9.0.115.0. */
        tosend[0] = 6;
        tosend[5] = 128;
        tosend[6] = 0;
        tosend[7] = 3;
        tosend[8] = 2;
        /* Initialize the Diffie-Hellmann context and generate the public key
         * to send to the server. */
        if ((ret = ff_rtmpe_gen_pub_key(rt->stream, tosend + 1)) < 0)
    client_pos = rtmp_handshake_imprint_with_digest(tosend + 1, rt->encrypted);
    if (client_pos < 0)
        return client_pos;
    if ((ret = ffurl_write(rt->stream, tosend,
                           RTMP_HANDSHAKE_PACKET_SIZE + 1)) < 0) {
        av_log(s, AV_LOG_ERROR, "Cannot write RTMP handshake request\n");
    if ((ret = ffurl_read_complete(rt->stream, serverdata,
                                   RTMP_HANDSHAKE_PACKET_SIZE + 1)) < 0) {
        av_log(s, AV_LOG_ERROR, "Cannot read RTMP handshake response\n");
    if ((ret = ffurl_read_complete(rt->stream, clientdata,
                                   RTMP_HANDSHAKE_PACKET_SIZE)) < 0) {
        av_log(s, AV_LOG_ERROR, "Cannot read RTMP handshake response\n");
    av_log(s, AV_LOG_DEBUG, "Type answer %d\n", serverdata[0]);
    av_log(s, AV_LOG_DEBUG, "Server version %d.%d.%d.%d\n",
           serverdata[5], serverdata[6], serverdata[7], serverdata[8]);
    if (rt->is_input && serverdata[5] >= 3) {
        server_pos = rtmp_validate_digest(serverdata + 1, 772);
        if (server_pos < 0)
            return server_pos;
        if (!server_pos) {
            type = 1;
            server_pos = rtmp_validate_digest(serverdata + 1, 8);
            if (server_pos < 0)
                return server_pos;
            if (!server_pos) {
                av_log(s, AV_LOG_ERROR, "Server response validating failed\n");
                return AVERROR(EIO);
        ret = ff_rtmp_calc_digest(tosend + 1 + client_pos, 32, 0,
                                  rtmp_server_key, sizeof(rtmp_server_key),
                                  digest);
        if (ret < 0)
        ret = ff_rtmp_calc_digest(clientdata, RTMP_HANDSHAKE_PACKET_SIZE - 32,
                                  0, digest, 32, signature);
        if (ret < 0)
        if (rt->encrypted && CONFIG_FFRTMPCRYPT_PROTOCOL) {
            /* Compute the shared secret key sent by the server and initialize
             * the RC4 encryption. */
            if ((ret = ff_rtmpe_compute_secret_key(rt->stream, serverdata + 1,
                                                   tosend + 1, type)) < 0)
            /* Encrypt the signature received by the server. */
            ff_rtmpe_encrypt_sig(rt->stream, signature, digest, serverdata[0]);
        if (memcmp(signature, clientdata + RTMP_HANDSHAKE_PACKET_SIZE - 32, 32)) {
            av_log(s, AV_LOG_ERROR, "Signature mismatch\n");
            return AVERROR(EIO);
        for (i = 0; i < RTMP_HANDSHAKE_PACKET_SIZE; i++)
            tosend[i] = av_lfg_get(&rnd) >> 24;
        ret = ff_rtmp_calc_digest(serverdata + 1 + server_pos, 32, 0,
                                  rtmp_player_key, sizeof(rtmp_player_key),
                                  digest);
        if (ret < 0)
        ret = ff_rtmp_calc_digest(tosend, RTMP_HANDSHAKE_PACKET_SIZE - 32, 0,
                                  digest, 32,
                                  tosend + RTMP_HANDSHAKE_PACKET_SIZE - 32);
        if (ret < 0)
        if (rt->encrypted && CONFIG_FFRTMPCRYPT_PROTOCOL) {
            /* Encrypt the signature to be send to the server. */
            ff_rtmpe_encrypt_sig(rt->stream, tosend +
                                 RTMP_HANDSHAKE_PACKET_SIZE - 32, digest,
                                 serverdata[0]);
        // write reply back to the server
        if ((ret = ffurl_write(rt->stream, tosend,
                               RTMP_HANDSHAKE_PACKET_SIZE)) < 0)
        if (rt->encrypted && CONFIG_FFRTMPCRYPT_PROTOCOL) {
            /* Set RC4 keys for encryption and update the keystreams. */
            if ((ret = ff_rtmpe_update_keystream(rt->stream)) < 0)
    } else {
        if (rt->encrypted && CONFIG_FFRTMPCRYPT_PROTOCOL) {
            /* Compute the shared secret key sent by the server and initialize
             * the RC4 encryption. */
            if ((ret = ff_rtmpe_compute_secret_key(rt->stream, serverdata + 1,
                            tosend + 1, 1)) < 0)
            if (serverdata[0] == 9) {
                /* Encrypt the signature received by the server. */
                ff_rtmpe_encrypt_sig(rt->stream, signature, digest,
                                     serverdata[0]);
        if ((ret = ffurl_write(rt->stream, serverdata + 1,
                               RTMP_HANDSHAKE_PACKET_SIZE)) < 0)
        if (rt->encrypted && CONFIG_FFRTMPCRYPT_PROTOCOL) {
            /* Set RC4 keys for encryption and update the keystreams. */
            if ((ret = ff_rtmpe_update_keystream(rt->stream)) < 0)
    return 0;