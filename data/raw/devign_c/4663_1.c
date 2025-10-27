static int ehci_reset_queue(EHCIQueue *q)

{

    int packets;



    trace_usb_ehci_queue_action(q, "reset");

    packets = ehci_cancel_queue(q);

    q->dev = NULL;

    q->qtdaddr = 0;


    return packets;

}