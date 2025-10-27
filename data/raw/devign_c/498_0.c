static void cpu_ioreq_pio(ioreq_t *req)

{

    int i;



    if (req->dir == IOREQ_READ) {

        if (!req->data_is_ptr) {

            req->data = do_inp(req->addr, req->size);

        } else {

            uint32_t tmp;



            for (i = 0; i < req->count; i++) {

                tmp = do_inp(req->addr, req->size);

                write_phys_req_item(req->data, req, i, &tmp);

            }

        }

    } else if (req->dir == IOREQ_WRITE) {

        if (!req->data_is_ptr) {

            do_outp(req->addr, req->size, req->data);

        } else {

            for (i = 0; i < req->count; i++) {

                uint32_t tmp = 0;



                read_phys_req_item(req->data, req, i, &tmp);

                do_outp(req->addr, req->size, tmp);

            }

        }

    }

}
