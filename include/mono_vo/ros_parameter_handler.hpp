
#pragma once

#include <rcl_interfaces/msg/parameter_descriptor.hpp>
#include <rclcpp/rclcpp.hpp>
#include <string>

namespace mono_vo
{
/**
 * @class RosParameterHandler
 * @brief A utility class to declare and retrieve ROS2 parameters for a component.
 *
 * This handler simplifies parameter management by combining declaration and retrieval
 * into a single call. The component class (e.g., Initializer) defines its member
 * variables with their default values, and this handler ensures they are properly
 * declared on the ROS node and updated from the parameter server (e.g., from a YAML file).
 */
class RosParameterHandler
{
public:
  RosParameterHandler(rclcpp::Node * node, const std::string & prefix = "")
  : node_(node), prefix_(prefix + "."), logger_(node->get_logger().get_child(prefix))
  {
  }

  /**
   * @brief Declares a parameter and retrieves its final value.
   *
   * This is the core function. It performs the following steps:
   * 1. Uses the initial value of `value_out` as the default for the declaration.
   * 2. Declares the parameter on the node with the given name and description.
   * 3. Retrieves the final value from the parameter server (which might have been
   *    overridden by a YAML file or command-line argument) and stores it back in `value_out`.
   *
   * @tparam T The type of the parameter.
   * @param name The short name of the parameter (e.g., "ransac_thresh").
   * @param value_out A reference to the member variable that holds the default value
   *                  and will be updated with the final value.
   * @param description An optional description for the parameter.
   */
  template <typename T>
  void declare_and_get(
    const std::string & name, T & value_out, const std::string & description = "")
  {
    // The initial value of the variable is used as the default
    const T default_value = value_out;
    const std::string full_name = prefix_ + name;

    // Create a parameter descriptor
    auto descriptor = rcl_interfaces::msg::ParameterDescriptor();
    descriptor.description = description;

    // Declare the parameter on the node
    node_->declare_parameter(full_name, default_value, descriptor);

    // Get the final value from the parameter server
    node_->get_parameter(full_name, value_out);

    // Log the result for easy debugging
    RCLCPP_INFO(
      logger_, "Parameter '%s' (default: %s) loaded with value: %s", name.c_str(),
      std::to_string(default_value).c_str(), std::to_string(value_out).c_str());
  }

private:
  rclcpp::Node * node_;
  std::string prefix_;
  rclcpp::Logger logger_;
};

}  // namespace mono_vo
