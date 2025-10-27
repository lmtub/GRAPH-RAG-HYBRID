static void cpu_ioreq_move(ioreq_t *req)

{

    int i;



    if (!req->data_is_ptr) {

        if (req->dir == IOREQ_READ) {

            for (i = 0; i < req->count; i++) {

                read_phys_req_item(req->addr, req, i, &req->data);

            }

        } else if (req->dir == IOREQ_WRITE) {

            for (i = 0; i < req->count; i++) {

                write_phys_req_item(req->addr, req, i, &req->data);

            }

        }

    } else {

        uint64_t tmp;



        if (req->dir == IOREQ_READ) {

            for (i = 0; i < req->count; i++) {

                read_phys_req_item(req->addr, req, i, &tmp);

                write_phys_req_item(req->data, req, i, &tmp);

            }

        } else if (req->dir == IOREQ_WRITE) {

            for (i = 0; i < req->count; i++) {

                read_phys_req_item(req->data, req, i, &tmp);

                write_phys_req_item(req->addr, req, i, &tmp);

            }

        }

    }

}
