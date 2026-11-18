.DEFAULT_GOAL := all

CFLAGS ?= -O3
CPPFLAGS += -Iinclude

BUILD_DIR := build
RAYLIB := lib/libraylib.a
RAYLIB_LIBS := $(RAYLIB) \
	-framework Cocoa \
	-framework OpenGL \
	-framework IOKit \
	-framework CoreVideo \
	-framework CoreFoundation

PROGRAMS := $(addprefix $(BUILD_DIR)/, \
	xor letter_interpol number_interpol number_interpol_mt)

.PHONY: all build_xor build_letters build_numbers build_multi

all: $(PROGRAMS)

build_xor: $(BUILD_DIR)/xor
build_letters: $(BUILD_DIR)/letter_interpol
build_numbers: $(BUILD_DIR)/number_interpol
build_multi: $(BUILD_DIR)/number_interpol_mt

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/xor: src/xor.c
$(BUILD_DIR)/letter_interpol: src/letter_interpolation.c $(RAYLIB)
$(BUILD_DIR)/number_interpol $(BUILD_DIR)/number_interpol_mt: src/number_interpolation.c $(RAYLIB)

$(BUILD_DIR)/number_interpol_mt: CPPFLAGS += -DNN_MULTITHREAD
$(filter-out $(BUILD_DIR)/xor,$(PROGRAMS)): LDLIBS += $(RAYLIB_LIBS)

$(PROGRAMS): | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -MMD -MP -MF $@.d -MT $@ \
		$(LDFLAGS) $< -o $@ $(LDLIBS)

-include $(addsuffix .d,$(PROGRAMS))
