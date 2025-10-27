uint16_t net_checksum_finish(uint32_t sum)

{

    while (sum>>16)

	sum = (sum & 0xFFFF)+(sum >> 16);

    return ~sum;

}
