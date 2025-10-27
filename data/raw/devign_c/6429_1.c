void qmp_memchar_write(const char *device, const char *data,
                       bool has_format, enum DataFormat format,
                       Error **errp)
{
    CharDriverState *chr;
    const uint8_t *write_data;
    int ret;
    size_t write_count;
    chr = qemu_chr_find(device);
    if (!chr) {
        error_setg(errp, "Device '%s' not found", device);
        return;
    if (qemu_is_chr(chr, "memory")) {
        error_setg(errp,"%s is not memory char device", device);
        return;
    if (has_format && (format == DATA_FORMAT_BASE64)) {
        write_data = g_base64_decode(data, &write_count);
    } else {
        write_data = (uint8_t *)data;
        write_count = strlen(data);
    ret = cirmem_chr_write(chr, write_data, write_count);
    if (ret < 0) {
        error_setg(errp, "Failed to write to device %s", device);
        return;