static void stm32f2xx_usart_write(void *opaque, hwaddr addr,

                                  uint64_t val64, unsigned int size)

{

    STM32F2XXUsartState *s = opaque;

    uint32_t value = val64;

    unsigned char ch;



    DB_PRINT("Write 0x%" PRIx32 ", 0x%"HWADDR_PRIx"\n", value, addr);



    switch (addr) {

    case USART_SR:

        if (value <= 0x3FF) {

            s->usart_sr = value;

        } else {

            s->usart_sr &= value;

        }

        if (!(s->usart_sr & USART_SR_RXNE)) {

            qemu_set_irq(s->irq, 0);

        }

        return;

    case USART_DR:

        if (value < 0xF000) {

            ch = value;

            if (s->chr) {



                qemu_chr_fe_write_all(s->chr, &ch, 1);

            }

            s->usart_sr |= USART_SR_TC;

            s->usart_sr &= ~USART_SR_TXE;

        }

        return;

    case USART_BRR:

        s->usart_brr = value;

        return;

    case USART_CR1:

        s->usart_cr1 = value;

            if (s->usart_cr1 & USART_CR1_RXNEIE &&

                s->usart_sr & USART_SR_RXNE) {

                qemu_set_irq(s->irq, 1);

            }

        return;

    case USART_CR2:

        s->usart_cr2 = value;

        return;

    case USART_CR3:

        s->usart_cr3 = value;

        return;

    case USART_GTPR:

        s->usart_gtpr = value;

        return;

    default:

        qemu_log_mask(LOG_GUEST_ERROR,

                      "%s: Bad offset 0x%"HWADDR_PRIx"\n", __func__, addr);

    }

}