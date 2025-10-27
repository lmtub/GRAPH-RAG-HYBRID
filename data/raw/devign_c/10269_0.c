static int tpm_passthrough_unix_tx_bufs(int tpm_fd,

                                        const uint8_t *in, uint32_t in_len,

                                        uint8_t *out, uint32_t out_len)

{

    int ret;



    ret = tpm_passthrough_unix_write(tpm_fd, in, in_len);

    if (ret != in_len) {

        error_report("tpm_passthrough: error while transmitting data "

                     "to TPM: %s (%i)\n",

                     strerror(errno), errno);

        goto err_exit;

    }



    ret = tpm_passthrough_unix_read(tpm_fd, out, out_len);

    if (ret < 0) {

        error_report("tpm_passthrough: error while reading data from "

                     "TPM: %s (%i)\n",

                     strerror(errno), errno);

    } else if (ret < sizeof(struct tpm_resp_hdr) ||

               tpm_passthrough_get_size_from_buffer(out) != ret) {

        ret = -1;

        error_report("tpm_passthrough: received invalid response "

                     "packet from TPM\n");

    }



err_exit:

    if (ret < 0) {

        tpm_write_fatal_error_response(out, out_len);

    }



    return ret;

}
