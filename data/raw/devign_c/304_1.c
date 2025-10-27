static size_t header_ext_add(char *buf, uint32_t magic, const void *s,

    size_t len, size_t buflen)

{

    QCowExtension *ext_backing_fmt = (QCowExtension*) buf;

    size_t ext_len = sizeof(QCowExtension) + ((len + 7) & ~7);



    if (buflen < ext_len) {

        return -ENOSPC;

    }



    *ext_backing_fmt = (QCowExtension) {

        .magic  = cpu_to_be32(magic),

        .len    = cpu_to_be32(len),

    };

    memcpy(buf + sizeof(QCowExtension), s, len);



    return ext_len;

}
