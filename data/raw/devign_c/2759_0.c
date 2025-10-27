static uint16_t mlp_checksum16(const uint8_t *buf, unsigned int buf_size)

{

    uint16_t crc;



    if (!crc_init) {

        av_crc_init(crc_2D, 0, 16, 0x002D, sizeof(crc_2D));

        crc_init = 1;

    }



    crc = av_crc(crc_2D, 0, buf, buf_size - 2);

    crc ^= AV_RL16(buf + buf_size - 2);

    return crc;

}
