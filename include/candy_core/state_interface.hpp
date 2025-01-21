#ifndef CANDY_STATE_INTERFACE_HPP
#define CANDY_STATE_INTERFACE_HPP

#include <string>
#include <memory>

namespace candy {

class StateInterface {
public:
  virtual ~StateInterface() = default;

  virtual void save_state(const std::string &key, const std::string &value) = 0;
  virtual std::string get_state(const std::string &key) const = 0;
};

} // namespace candy

#endif // CANDY_STATE_INTERFACE_HPP
