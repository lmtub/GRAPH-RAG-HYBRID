static void smbios_encode_uuid(struct smbios_uuid *uuid, const uint8_t *buf)

{

    memcpy(uuid, buf, 16);

    if (smbios_uuid_encoded) {

        uuid->time_low = bswap32(uuid->time_low);

        uuid->time_mid = bswap16(uuid->time_mid);

        uuid->time_hi_and_version = bswap16(uuid->time_hi_and_version);

    }

}
