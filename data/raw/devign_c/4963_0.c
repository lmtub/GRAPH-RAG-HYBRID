static int qcow2_probe(const uint8_t *buf, int buf_size, const char *filename)

{

    const QCowHeader *cow_header = (const void *)buf;



    if (buf_size >= sizeof(QCowHeader) &&

        be32_to_cpu(cow_header->magic) == QCOW_MAGIC &&

        be32_to_cpu(cow_header->version) >= QCOW_VERSION)

        return 100;

    else

        return 0;

}
