static void piix4_reset(void *opaque)

{

	PIIX4PMState *s = opaque;

	uint8_t *pci_conf = s->dev.config;



	pci_conf[0x58] = 0;

	pci_conf[0x59] = 0;

	pci_conf[0x5a] = 0;

	pci_conf[0x5b] = 0;

}
