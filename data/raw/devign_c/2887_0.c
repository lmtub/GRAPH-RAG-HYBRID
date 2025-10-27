static int acpi_checksum(const uint8_t *data, int len)

{

    int sum, i;

    sum = 0;

    for(i = 0; i < len; i++)

        sum += data[i];

    return (-sum) & 0xff;

}
