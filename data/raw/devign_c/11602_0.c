putsum(uint8_t *data, uint32_t n, uint32_t sloc, uint32_t css, uint32_t cse)

{

    if (cse && cse < n)

        n = cse + 1;

    if (sloc < n-1)

        cpu_to_be16wu((uint16_t *)(data + sloc),

                      do_cksum(data + css, data + n));

}
