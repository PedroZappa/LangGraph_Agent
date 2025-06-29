#==============================================================================#
#                                  MAKE CONFIG                                 #
#==============================================================================#

MAKE	= make --no-print-directory
SHELL	:= bash --rcfile ~/.bashrc

#==============================================================================#
#                                     NAMES                                    #
#==============================================================================#

NAME = langgraph_agent
MAIN = main.py
SRC = .
ARGS = 
VENV = .venv
MAIN_TEST = test_$(NAME).py
EXEC = ./scripts/run.sh && python3 $(MAIN)

### Message Vars
_SUCCESS			= [$(GRN)SUCCESS$(D)]
_INFO					= [$(BLU)INFO$(D)]
_NORM_SUCCESS = $(GRN)=== OK:$(D)
_NORM_INFO 		= $(BLU)File no:$(D)
_NORM_ERR 		= $(RED)=== KO:$(D)
_SEP					= =====================

# Test summary box components
TEST_BOX_TOP	:= ╔═══════════════════════════════════╗
TEST_BOX_MID	:= ╠═══════════════════════════════════╣
TEST_BOX_BOT	:= ╚═══════════════════════════════════╝
TEST_HEADER		:= ║           TEST SUMMARY            ║

#==============================================================================#
#                                COMMANDS                                      #
#==============================================================================#

### Core Utils
RM		= rm -rf
MKDIR_P	= mkdir -p

#==============================================================================#
#                                  RULES                                       #
#==============================================================================#

##@ Project Scaffolding 󰛵

all:			## Build and run project
	$(MAKE) run

run:			## Run project
	@if [ ! -d "$(VENV)" ]; then \
		$(MAKE) build; \
	fi
	@echo "* $(MAG)$(NAME) $(YEL)executing$(D): $(_SUCCESS)"
	@echo "$(GRN)$(_SEP)$(D)"
	@source $(EXEC) $(ARGS)
	@echo "$(GRN)$(_SEP)$(D)"
	@echo "* $(MAG)$(NAME) $(YEL)finished$(D):"

build:		## Build project
	@echo "* $(MAG)$(NAME) $(YEL)building$(D): $(_SUCCESS)"
	source ./scripts/build.sh
	@echo "* $(MAG)$(NAME) $(YEL)finished building$(D):"

##@ Utility Rules 

# Define a function to add directory if it exists
add-if-exists = $(if $(wildcard $(1)/.),$(1))

EXCLUDE_DIRS = $(VENV) \
               $(call add-if-exists,__*) \
               $(call add-if-exists,.*_cache) \

black: ## Run black formatter
	black . --exclude=$(EXCLUDE_DIRS)

sphinx:		## Generate .rst files
	sphinx-apidoc -o docs/source $(SRC)
	# sphinx-build -M html docs/source/ docs/build/
	$(MAKE) -C docs html

docs: 		## Open docs index in browser
	xdg-open docs/build/html/index.html

##@ Test/Debug Rules 

test:## Run all tests
	@echo "* $(MAG)$(NAME) $(YEL)starting test suite$(D):"
	@echo ""
	@$(MAKE) doctest; DOCTEST_EXIT=$$?; \
	$(MAKE) pytest; PYTEST_EXIT=$$?; \
	$(MAKE) mypy; MYPY_EXIT=$$?; \
	echo ""; \
	echo "$(MAG)$(TEST_BOX_TOP)$(D)"; \
	echo "$(MAG)$(TEST_HEADER)$(D)"; \
	echo "$(MAG)$(TEST_BOX_MID)$(D)"; \
	print_test_result() { \
		test_name="$$2"; \
		name_length=$${#test_name}; \
		if [ $$1 -eq 0 ]; then \
			status="$(GRN)PASSED$(D)"; \
			icon="$(GRN)✓$(D)"; \
		else \
			status="$(RED)FAILED$(D)"; \
			icon="$(RED)✗$(D)"; \
		fi; \
		padding=$$((25 - name_length)); \
		printf "$(MAG)║$(D) %s %s %s%*s$(MAG)║$(D)\n" \
			"$$icon" "$$test_name" "$$status" "$$padding" ""; \
	}; \
	print_test_result $$DOCTEST_EXIT "Doctest"; \
	print_test_result $$PYTEST_EXIT "Pytest"; \
	print_test_result $$MYPY_EXIT "MyPy"; \
	echo "$(MAG)$(TEST_BOX_BOT)$(D)"; \
	echo ""; \
	TOTAL_FAILED=$$(($$DOCTEST_EXIT + $$PYTEST_EXIT + $$MYPY_EXIT)); \
	if [ $$TOTAL_FAILED -eq 0 ]; then \
		echo "$(GRN)🎉 All tests passed successfully!$(D)"; \
		exit 0; \
	else \
		echo "$(RED)❌ $$TOTAL_FAILED test suite(s) failed$(D)"; \
		exit 1; \
	fi

doctest: sphinx		## Run sphinx doctests
	$(MAKE) -C docs doctest

pytest:		## run pytest
	@echo "* $(MAG)$(NAME) $(YEL)running pytest$(D):"
	pytest $(MAIN_TEST)
	@echo "* $(MAG)$(NAME) pytest suite $(YEL)finished$(D):"

MYPY_FLAGS := --install-types --ignore-missing-imports

mypy:			## Run mypy static checker
	@echo "* $(MAG)$(NAME) $(YEL)running type checker$(D):"
	mypy $(MYPY_FLAGS) $(MAIN)
	@echo "* $(MAG)$(NAME) type checker $(YEL)finished$(D):"

posting:	## Run posting API testing client
	posting --collection $(NAME)_posting --env .env

##@ Clean-up Rules 󰃢

# Define files/directories to clean
CLEAN_TARGETS := $(EXCLUDE_DIRS) \
                 *.sqlite \
                 *.pyc \
                 *_dump.csv

clean: ## Remove object files
	@echo "*** $(YEL)Removing $(MAG)$(NAME)$(D) and deps $(YEL)object files$(D)"
	@for target in $(CLEAN_TARGETS); do \
		if [ -e "$$target" ] || [ -d "$$target" ]; then \
			$(RM) "$$target"; \
			echo "*** $(YEL)Removed $(CYA)$$target$(D)"; \
		fi; \
	done
	@echo "$(_SUCCESS) Clean completed!"

##@ Help 󰛵

help: 	## Display this help page
	@awk 'BEGIN {FS = ":.*##"; \
			printf "\n=> Usage:\n\tmake $(GRN)<target>$(D)\n"} \
		/^[a-zA-Z_0-9-]+:.*?##/ { \
			printf "\t$(GRN)%-18s$(D) %s\n", $$1, $$2 } \
		/^##@/ { \
			printf "\n=> %s\n", substr($$0, 5) } ' Makefile
## Tweaked from source:
### https://www.padok.fr/en/blog/beautiful-makefile-awk

.PHONY: test mypy black posting clean help docs

#==============================================================================#
#                                  UTILS                                       #
#==============================================================================#

# Colors
#
# Run the following command to get list of available colors
# bash -c 'for c in {0..255}; do tput setaf $c; tput setaf $c | cat -v; echo =$c; done'
#
B  		= $(shell tput bold)
BLA		= $(shell tput setaf 0)
RED		= $(shell tput setaf 1)
GRN		= $(shell tput setaf 2)
YEL		= $(shell tput setaf 3)
BLU		= $(shell tput setaf 4)
MAG		= $(shell tput setaf 5)
CYA		= $(shell tput setaf 6)
WHI		= $(shell tput setaf 7)
GRE		= $(shell tput setaf 8)
BRED 	= $(shell tput setaf 9)
BGRN	= $(shell tput setaf 10)
BYEL	= $(shell tput setaf 11)
BBLU	= $(shell tput setaf 12)
BMAG	= $(shell tput setaf 13)
BCYA	= $(shell tput setaf 14)
BWHI	= $(shell tput setaf 15)
D 		= $(shell tput sgr0)
BEL 	= $(shell tput bel)
CLR 	= $(shell tput el 1)






