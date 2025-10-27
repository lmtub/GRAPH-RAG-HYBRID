int ff_srtp_decrypt(struct SRTPContext *s, uint8_t *buf, int *lenptr)

{

    uint8_t iv[16] = { 0 }, hmac[20];

    int len = *lenptr;

    int ext, seq_largest;

    uint32_t ssrc, roc;

    uint64_t index;

    int rtcp;



    // TODO: Missing replay protection



    if (len < s->hmac_size)

        return AVERROR_INVALIDDATA;



    rtcp = RTP_PT_IS_RTCP(buf[1]);



    // Authentication HMAC

    av_hmac_init(s->hmac, rtcp ? s->rtcp_auth : s->rtp_auth, sizeof(s->rtp_auth));

    // If MKI is used, this should exclude the MKI as well

    av_hmac_update(s->hmac, buf, len - s->hmac_size);



    if (!rtcp) {

        int seq = AV_RB16(buf + 2);

        uint32_t v;

        uint8_t rocbuf[4];



        // RFC 3711 section 3.3.1, appendix A

        seq_largest = s->seq_initialized ? s->seq_largest : seq;

        v = roc = s->roc;

        if (seq_largest < 32768) {

            if (seq - seq_largest > 32768)

                v = roc - 1;

        } else {

            if (seq_largest - 32768 > seq)

                v = roc + 1;

        }

        if (v == roc) {

            seq_largest = FFMAX(seq_largest, seq);

        } else if (v == roc + 1) {

            seq_largest = seq;

            roc = v;

        }

        index = seq + (((uint64_t)v) << 16);



        AV_WB32(rocbuf, roc);

        av_hmac_update(s->hmac, rocbuf, 4);

    }



    av_hmac_final(s->hmac, hmac, sizeof(hmac));

    if (memcmp(hmac, buf + len - s->hmac_size, s->hmac_size)) {

        av_log(NULL, AV_LOG_WARNING, "HMAC mismatch\n");

        return AVERROR_INVALIDDATA;

    }



    len -= s->hmac_size;

    *lenptr = len;



    if (len < 12)

        return AVERROR_INVALIDDATA;



    if (rtcp) {

        uint32_t srtcp_index = AV_RB32(buf + len - 4);

        len -= 4;

        *lenptr = len;



        ssrc = AV_RB32(buf + 4);

        index = srtcp_index & 0x7fffffff;



        buf += 8;

        len -= 8;

        if (!(srtcp_index & 0x80000000))

            return 0;

    } else {

        s->seq_initialized = 1;

        s->seq_largest     = seq_largest;

        s->roc             = roc;



        ext  = buf[0] & 0x10;

        ssrc = AV_RB32(buf + 8);



        buf += 12;

        len -= 12;



        if (ext) {

            if (len < 4)

                return AVERROR_INVALIDDATA;

            ext = (AV_RB16(buf + 2) + 1) * 4;

            if (len < ext)

                return AVERROR_INVALIDDATA;

            len -= ext;

            buf += ext;

        }

    }



    create_iv(iv, rtcp ? s->rtcp_salt : s->rtp_salt, index, ssrc);

    av_aes_init(s->aes, rtcp ? s->rtcp_key : s->rtp_key, 128, 0);

    encrypt_counter(s->aes, iv, buf, len);



    return 0;

}
