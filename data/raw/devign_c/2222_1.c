static void icp_set_cppr(struct icp_state *icp, int server, uint8_t cppr)

{

    struct icp_server_state *ss = icp->ss + server;

    uint8_t old_cppr;

    uint32_t old_xisr;



    old_cppr = CPPR(ss);

    ss->xirr = (ss->xirr & ~CPPR_MASK) | (cppr << 24);



    if (cppr < old_cppr) {

        if (XISR(ss) && (cppr <= ss->pending_priority)) {

            old_xisr = XISR(ss);

            ss->xirr &= ~XISR_MASK; /* Clear XISR */


            qemu_irq_lower(ss->output);

            ics_reject(icp->ics, old_xisr);

        }

    } else {

        if (!XISR(ss)) {

            icp_resend(icp, server);

        }

    }

}