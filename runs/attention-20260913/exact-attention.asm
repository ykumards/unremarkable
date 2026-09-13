0000f4e4 <unremarkable::causal_attention(float const*, float const*, float const*, int, int, int, int, int, float*, float*)>:
    f4e4: e92d 4ff0    	push.w	{r4, r5, r6, r7, r8, r9, r10, r11, lr}
    f4e8: 461f         	mov	r7, r3
    f4ea: ed2d 8b04    	vpush	{d8, d9}
    f4ee: b091         	sub	sp, #0x44
    f4f0: 9d1e         	ldr	r5, [sp, #0x78]
    f4f2: e9dd 641f    	ldrd	r6, r4, [sp, #124]
    f4f6: 9306         	str	r3, [sp, #0x18]
    f4f8: fb06 f305    	mul	r3, r6, r5
    f4fc: 9009         	str	r0, [sp, #0x24]
    f4fe: 4638         	mov	r0, r7
    f500: e9cd 120d    	strd	r1, r2, [sp, #52]
    f504: 9300         	str	r3, [sp]
    f506: 4629         	mov	r1, r5
    f508: f8dd a084    	ldr.w	r10, [sp, #0x84]
    f50c: f07c f800    	bl	0x8b510 <__divsi3>      @ imm = #0x7c000
    f510: ee07 6a90    	vmov	s15, r6
    f514: 9b00         	ldr	r3, [sp]
    f516: 9007         	str	r0, [sp, #0x1c]
    f518: eeb8 0ae7    	vcvt.f32.s32	s0, s15
    f51c: eeb5 0a40    	vcmp.f32	s0, #0
    f520: eef1 fa10    	vmrs	APSR_nzcv, fpscr
    f524: f100 80c2    	bmi.w	0xf6ac <unremarkable::causal_attention(float const*, float const*, float const*, int, int, int, int, int, float*, float*)+0x1c8> @ imm = #0x184
    f528: eeb1 9ac0    	vsqrt.f32	s18, s0
    f52c: 9a06         	ldr	r2, [sp, #0x18]
    f52e: 2a00         	cmp	r2, #0x0
    f530: f340 80ad    	ble.w	0xf68e <unremarkable::causal_attention(float const*, float const*, float const*, int, int, int, int, int, float*, float*)+0x1aa> @ imm = #0x15a
    f534: 00a2         	lsls	r2, r4, #0x2
    f536: 1ef5         	subs	r5, r6, #0x3
    f538: 9208         	str	r2, [sp, #0x20]
    f53a: 1f32         	subs	r2, r6, #0x4
    f53c: 9922         	ldr	r1, [sp, #0x88]
    f53e: f022 0203    	bic	r2, r2, #0x3
    f542: 9809         	ldr	r0, [sp, #0x24]
    f544: 3204         	adds	r2, #0x4
    f546: f8dd 808c    	ldr.w	r8, [sp, #0x8c]
    f54a: eb01 098a    	add.w	r9, r1, r10, lsl #2
    f54e: 920c         	str	r2, [sp, #0x30]
    f550: f109 0904    	add.w	r9, r9, #0x4
    f554: 00b2         	lsls	r2, r6, #0x2
    f556: 1884         	adds	r4, r0, r2
    f558: eb08 0702    	add.w	r7, r8, r2
    f55c: 9203         	str	r2, [sp, #0xc]
    f55e: 2200         	movs	r2, #0x0
    f560: 9100         	str	r1, [sp]
    f562: 9202         	str	r2, [sp, #0x8]
    f564: 0099         	lsls	r1, r3, #0x2
    f566: 9201         	str	r2, [sp, #0x4]
    f568: f10a 0201    	add.w	r2, r10, #0x1
    f56c: 910b         	str	r1, [sp, #0x2c]
    f56e: 920a         	str	r2, [sp, #0x28]
    f570: 930f         	str	r3, [sp, #0x3c]
    f572: 9907         	ldr	r1, [sp, #0x1c]
    f574: 9801         	ldr	r0, [sp, #0x4]
    f576: f07b ffcb    	bl	0x8b510 <__divsi3>      @ imm = #0x7bf96
    f57a: f1ba 0f00    	cmp.w	r10, #0x0
    f57e: fb06 fb00    	mul	r11, r6, r0
    f582: f2c0 8089    	blt.w	0xf698 <unremarkable::causal_attention(float const*, float const*, float const*, int, int, int, int, int, float*, float*)+0x1b4> @ imm = #0x112
    f586: 9b09         	ldr	r3, [sp, #0x24]
    f588: 46dc         	mov	r12, r11
    f58a: 9a02         	ldr	r2, [sp, #0x8]
    f58c: f8cd b010    	str.w	r11, [sp, #0x10]
    f590: eb03 0e82    	add.w	lr, r3, r2, lsl #2
    f594: 9800         	ldr	r0, [sp]
    f596: 9b0f         	ldr	r3, [sp, #0x3c]
    f598: f8dd b034    	ldr.w	r11, [sp, #0x34]
    f59c: eddf 7a47    	vldr	s15, [pc, #284]         @ 0xf6bc <unremarkable::causal_attention(float const*, float const*, float const*, int, int, int, int, int, float*, float*)+0x1d8>
    f5a0: 2e00         	cmp	r6, #0x0
    f5a2: dd0a         	ble	0xf5ba <unremarkable::causal_attention(float const*, float const*, float const*, int, int, int, int, int, float*, float*)+0xd6> @ imm = #0x14
    f5a4: eb0b 018c    	add.w	r1, r11, r12, lsl #2
    f5a8: 4672         	mov	r2, lr
    f5aa: ecf2 6a01    	vldmia	r2!, {s13}
    f5ae: ecb1 7a01    	vldmia	r1!, {s14}
    f5b2: ee46 7a87    	vmla.f32	s15, s13, s14
    f5b6: 42a2         	cmp	r2, r4
    f5b8: d1f7         	bne	0xf5aa <unremarkable::causal_attention(float const*, float const*, float const*, int, int, int, int, int, float*, float*)+0xc6> @ imm = #-0x12
    f5ba: ee87 7a89    	vdiv.f32	s14, s15, s18
    f5be: 449c         	add	r12, r3
    f5c0: eca0 7a01    	vstmia	r0!, {s14}
    f5c4: 4548         	cmp	r0, r9
    f5c6: d1e9         	bne	0xf59c <unremarkable::causal_attention(float const*, float const*, float const*, int, int, int, int, int, float*, float*)+0xb8> @ imm = #-0x2e
    f5c8: 990a         	ldr	r1, [sp, #0x28]
    f5ca: 9800         	ldr	r0, [sp]
    f5cc: f8dd b010    	ldr.w	r11, [sp, #0x10]
    f5d0: f7ff fb86    	bl	0xece0 <unremarkable::softmax_inplace(float*, int)> @ imm = #-0x8f4
    f5d4: 9a03         	ldr	r2, [sp, #0xc]
    f5d6: 2100         	movs	r1, #0x0
    f5d8: 4640         	mov	r0, r8
    f5da: f7f8 eb26    	blx	0x7c28 <memset@plt>     @ imm = #-0x79b4
    f5de: 9b0e         	ldr	r3, [sp, #0x38]
    f5e0: 2e03         	cmp	r6, #0x3
    f5e2: f8cd 9010    	str.w	r9, [sp, #0x10]
    f5e6: f04f 0e00    	mov.w	lr, #0x0
    f5ea: eb03 0c8b    	add.w	r12, r3, r11, lsl #2
    f5ee: f8dd b000    	ldr.w	r11, [sp]
    f5f2: 9405         	str	r4, [sp, #0x14]
    f5f4: f8dd 9030    	ldr.w	r9, [sp, #0x30]
    f5f8: 9c0b         	ldr	r4, [sp, #0x2c]
    f5fa: f85b 0b04    	ldr	r0, [r11], #4
    f5fe: dd2f         	ble	0xf660 <unremarkable::causal_attention(float const*, float const*, float const*, int, int, int, int, int, float*, float*)+0x17c> @ imm = #0x5e
    f600: 4642         	mov	r2, r8
    f602: 4661         	mov	r1, r12
    f604: 2300         	movs	r3, #0x0
    f606: f961 2a8d    	vld1.32	{d18, d19}, [r1]!
    f60a: ee08 0b10    	vmov.32	d8[0], r0
    f60e: f962 0a8f    	vld1.32	{d16, d17}, [r2]
    f612: ffe2 29c8    	vmul.f32	q9, q9, d8[0]
    f616: 3304         	adds	r3, #0x4
    f618: 42ab         	cmp	r3, r5
    f61a: ef40 0de2    	vadd.f32	q8, q8, q9
    f61e: f942 0a8d    	vst1.32	{d16, d17}, [r2]!
    f622: dbf0         	blt	0xf606 <unremarkable::causal_attention(float const*, float const*, float const*, int, int, int, int, int, float*, float*)+0x122> @ imm = #-0x20
    f624: 464a         	mov	r2, r9
    f626: 4296         	cmp	r6, r2
    f628: dd10         	ble	0xf64c <unremarkable::causal_attention(float const*, float const*, float const*, int, int, int, int, int, float*, float*)+0x168> @ imm = #0x20
    f62a: 0092         	lsls	r2, r2, #0x2
    f62c: 4601         	mov	r1, r0
    f62e: eb08 0302    	add.w	r3, r8, r2
    f632: 4462         	add	r2, r12
    f634: ee06 1a90    	vmov	s13, r1
    f638: edd3 7a00    	vldr	s15, [r3]
    f63c: ecb2 7a01    	vldmia	r2!, {s14}
    f640: ee47 7a26    	vmla.f32	s15, s14, s13
    f644: ece3 7a01    	vstmia	r3!, {s15}
    f648: 42bb         	cmp	r3, r7
    f64a: d1f3         	bne	0xf634 <unremarkable::causal_attention(float const*, float const*, float const*, int, int, int, int, int, float*, float*)+0x150> @ imm = #-0x1a
    f64c: 45f2         	cmp	r10, lr
    f64e: f10e 0301    	add.w	r3, lr, #0x1
    f652: 44a4         	add	r12, r4
    f654: d006         	beq	0xf664 <unremarkable::causal_attention(float const*, float const*, float const*, int, int, int, int, int, float*, float*)+0x180> @ imm = #0xc
    f656: f85b 0b04    	ldr	r0, [r11], #4
    f65a: 2e03         	cmp	r6, #0x3
    f65c: 469e         	mov	lr, r3
    f65e: dccf         	bgt	0xf600 <unremarkable::causal_attention(float const*, float const*, float const*, int, int, int, int, int, float*, float*)+0x11c> @ imm = #-0x62
    f660: 2200         	movs	r2, #0x0
    f662: e7e0         	b	0xf626 <unremarkable::causal_attention(float const*, float const*, float const*, int, int, int, int, int, float*, float*)+0x142> @ imm = #-0x40
    f664: e9dd 9404    	ldrd	r9, r4, [sp, #16]
    f668: 9908         	ldr	r1, [sp, #0x20]
    f66a: 9b00         	ldr	r3, [sp]
    f66c: 9a03         	ldr	r2, [sp, #0xc]
    f66e: 440b         	add	r3, r1
    f670: 9300         	str	r3, [sp]
    f672: 9b01         	ldr	r3, [sp, #0x4]
    f674: 4490         	add	r8, r2
    f676: 4417         	add	r7, r2
    f678: 3301         	adds	r3, #0x1
    f67a: 4414         	add	r4, r2
    f67c: 9a06         	ldr	r2, [sp, #0x18]
    f67e: 4489         	add	r9, r1
    f680: 9902         	ldr	r1, [sp, #0x8]
    f682: 429a         	cmp	r2, r3
    f684: 4431         	add	r1, r6
    f686: 9301         	str	r3, [sp, #0x4]
    f688: 9102         	str	r1, [sp, #0x8]
    f68a: f47f af72    	bne.w	0xf572 <unremarkable::causal_attention(float const*, float const*, float const*, int, int, int, int, int, float*, float*)+0x8e> @ imm = #-0x11c
    f68e: b011         	add	sp, #0x44
    f690: ecbd 8b04    	vpop	{d8, d9}
    f694: e8bd 8ff0    	pop.w	{r4, r5, r6, r7, r8, r9, r10, r11, pc}
    f698: 990a         	ldr	r1, [sp, #0x28]
    f69a: 9800         	ldr	r0, [sp]
    f69c: f7ff fb20    	bl	0xece0 <unremarkable::softmax_inplace(float*, int)> @ imm = #-0x9c0
    f6a0: 9a03         	ldr	r2, [sp, #0xc]
    f6a2: 2100         	movs	r1, #0x0
    f6a4: 4640         	mov	r0, r8
    f6a6: f7f8 eac0    	blx	0x7c28 <memset@plt>     @ imm = #-0x7a80
    f6aa: e7dd         	b	0xf668 <unremarkable::causal_attention(float const*, float const*, float const*, int, int, int, int, int, float*, float*)+0x184> @ imm = #-0x46
    f6ac: 9300         	str	r3, [sp]
    f6ae: f7f8 ea60    	blx	0x7b70 <sqrtf@plt>      @ imm = #-0x7b40
    f6b2: eeb0 9a40    	vmov.f32	s18, s0
    f6b6: 9b00         	ldr	r3, [sp]
    f6b8: e738         	b	0xf52c <unremarkable::causal_attention(float const*, float const*, float const*, int, int, int, int, int, float*, float*)+0x48> @ imm = #-0x190
    f6ba: bf00         	nop
    f6bc: 00 00 00 00  	.word	0x00000000

