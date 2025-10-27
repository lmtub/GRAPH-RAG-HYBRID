static int mxf_read_cryptographic_context(void *arg, AVIOContext *pb, int tag, int size, UID uid)

{

    MXFCryptoContext *cryptocontext = arg;

    if (size != 16)

        return -1;

    if (IS_KLV_KEY(uid, mxf_crypto_source_container_ul))

        avio_read(pb, cryptocontext->source_container_ul, 16);

    return 0;

}
