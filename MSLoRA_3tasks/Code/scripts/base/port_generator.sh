# port_generator.sh

generate_random_port() {
  local min_port=1024
  local max_port=65535
  while :; do
    port=$(shuf -i ${min_port}-${max_port} -n 1)
    if ! ss -ltn | awk '{print $4}' | grep -q ":$port$"; then
      echo $port
      return
    fi
  done
}