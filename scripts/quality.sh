#!/usr/bin/env bash
set -euo pipefail

# Quality gate for nucleon. Runs the checks every change must pass before
# it is called done (same convention as higgs/scripts/quality.sh):
#
#   * cargo fmt    (apply, then verify clean)
#   * cargo clippy --all-targets -- -D warnings
#   * cargo test --workspace
#
# Metal parity tests need a GPU; on a machine without one they skip
# visibly (a kernel compile error still fails). Coverage gates are
# separate and heavier; they arrive with scripts/coverage.sh once the
# engine has enough surface to gate.

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

step() { printf "${CYAN}==> %s${NC}\n" "$1"; }
ok()   { printf "${GREEN}OK  %s${NC}\n" "$1"; }
fail() { printf "${RED}FAIL %s${NC}\n" "$1"; exit 1; }

step "cargo fmt (apply + verify)"
cargo fmt --all
cargo fmt --all -- --check || fail "fmt left changes"
ok "fmt"

step "cargo clippy --all-targets -- -D warnings"
cargo clippy --all-targets -- -D warnings || fail "clippy"
ok "clippy"

step "cargo test --workspace"
cargo test --workspace || fail "tests"
ok "tests"

printf "${GREEN}quality gate passed${NC}\n"
