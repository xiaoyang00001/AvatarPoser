#include "TrackingInfoUdpBroadcaster.h"
#include "Logger.h"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#include <ws2tcpip.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

namespace {

const char* GetEnvOrDefault(const char* name, const char* defaultValue)
{
	const char* value = std::getenv(name);
	if (value == nullptr || value[0] == '\0') {
		return defaultValue;
	}
	return value;
}

int GetEnvIntOrDefault(const char* name, int defaultValue)
{
	const char* value = std::getenv(name);
	if (value == nullptr || value[0] == '\0') {
		return defaultValue;
	}
	return std::atoi(value);
}

bool IsDisabled()
{
	const char* value = std::getenv("AVATARPOSER_TRACKING_EXPORT");
	return value != nullptr && value[0] == '0';
}

SOCKET ToSocket(uintptr_t value)
{
	return static_cast<SOCKET>(value);
}

bool IsValidPosition(const TrackingVector3& position)
{
	if (!std::isfinite(position.x) || !std::isfinite(position.y) || !std::isfinite(position.z)) {
		return false;
	}
	if (position.x == 0.0f && position.y == 0.0f && position.z == 0.0f) {
		return false;
	}
	if (std::fabs(position.x) > 100.0f || std::fabs(position.y) > 100.0f || std::fabs(position.z) > 100.0f) {
		return false;
	}
	return true;
}

bool IsValidQuaternion(const TrackingQuat& rotation)
{
	if (!std::isfinite(rotation.x) || !std::isfinite(rotation.y) ||
		!std::isfinite(rotation.z) || !std::isfinite(rotation.w)) {
		return false;
	}

	const float norm = std::sqrt(
		rotation.x * rotation.x +
		rotation.y * rotation.y +
		rotation.z * rotation.z +
		rotation.w * rotation.w);
	return std::fabs(norm - 1.0f) <= 1e-3f;
}

bool IsValidPose(const TrackingVector3& position, const TrackingQuat& rotation)
{
	return IsValidPosition(position) && IsValidQuaternion(rotation);
}

} // namespace

TrackingInfoUdpBroadcaster& TrackingInfoUdpBroadcaster::Instance()
{
	static TrackingInfoUdpBroadcaster instance;
	return instance;
}

TrackingInfoUdpBroadcaster::TrackingInfoUdpBroadcaster()
	: m_initialized(false)
	, m_enabled(false)
	, m_host("127.0.0.1")
	, m_port(39001)
	, m_socket(0)
	, m_destinationAddress(0)
	, m_destinationPort(0)
	, m_lastBroadcastFrame(static_cast<uint64_t>(-1))
	, m_invalidFrameCount(0)
{
	Initialize();
}

TrackingInfoUdpBroadcaster::~TrackingInfoUdpBroadcaster()
{
	if (m_socket != 0) {
		closesocket(ToSocket(m_socket));
		m_socket = 0;
	}
	if (m_initialized) {
		WSACleanup();
	}
}

bool TrackingInfoUdpBroadcaster::Initialize()
{
	if (IsDisabled()) {
		return false;
	}

	m_host = GetEnvOrDefault("AVATARPOSER_TRACKING_HOST", "127.0.0.1");
	m_port = GetEnvIntOrDefault("AVATARPOSER_TRACKING_PORT", 39001);

	WSADATA wsaData;
	if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
		Log("AvatarPoser tracking exporter: WSAStartup failed.");
		return false;
	}
	m_initialized = true;

	SOCKET socketHandle = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
	if (socketHandle == INVALID_SOCKET) {
		Log("AvatarPoser tracking exporter: socket creation failed.");
		return false;
	}

	in_addr destinationAddress;
	if (inet_pton(AF_INET, m_host.c_str(), &destinationAddress) != 1) {
		Log("AvatarPoser tracking exporter: invalid host %s.", m_host.c_str());
		closesocket(socketHandle);
		return false;
	}

	m_destinationAddress = destinationAddress.s_addr;
	m_destinationPort = htons(static_cast<u_short>(m_port));
	m_socket = static_cast<uintptr_t>(socketHandle);
	m_enabled = true;
	Log("AvatarPoser tracking exporter: UDP %s:%d.", m_host.c_str(), m_port);
	return true;
}

void TrackingInfoUdpBroadcaster::Broadcast(const TrackingInfo& info)
{
	if (!m_enabled || m_socket == 0 || m_lastBroadcastFrame == info.FrameIndex) {
		return;
	}
	if (!IsTrackingInfoValid(info)) {
		++m_invalidFrameCount;
		if (m_invalidFrameCount == 1 || m_invalidFrameCount % 300 == 0) {
			Log("AvatarPoser tracking exporter: skipped invalid 6DoF tracking frame, skipped=%llu.",
				static_cast<unsigned long long>(m_invalidFrameCount));
		}
		m_lastBroadcastFrame = info.FrameIndex;
		return;
	}

	char payload[2048];
	int payloadLength = 0;
	BuildPayload(payload, sizeof(payload), payloadLength, info);
	if (payloadLength <= 0) {
		return;
	}

	sockaddr_in destination;
	std::memset(&destination, 0, sizeof(destination));
	destination.sin_family = AF_INET;
	destination.sin_port = m_destinationPort;
	destination.sin_addr.s_addr = m_destinationAddress;

	sendto(
		ToSocket(m_socket),
		payload,
		payloadLength,
		0,
		reinterpret_cast<const sockaddr*>(&destination),
		sizeof(destination));

	m_lastBroadcastFrame = info.FrameIndex;
}

bool TrackingInfoUdpBroadcaster::IsTrackingInfoValid(const TrackingInfo& info) const
{
	return IsValidPose(info.HeadPose_Pose_Position, info.HeadPose_Pose_Orientation) &&
		IsValidPose(info.LeftHand_Pose_Position, info.LeftHand_Pose_Orientation) &&
		IsValidPose(info.RightHand_Pose_Position, info.RightHand_Pose_Orientation);
}

void TrackingInfoUdpBroadcaster::BuildPayload(char* payload, size_t payloadSize, int& payloadLength, const TrackingInfo& info) const
{
	payloadLength = _snprintf_s(
		payload,
		payloadSize,
		_TRUNCATE,
		"{\"frame\":%llu,\"timestamp\":%.6f,"
		"\"head\":{\"quaternion\":[%.8f,%.8f,%.8f,%.8f],\"position\":[%.8f,%.8f,%.8f]},"
		"\"left_hand\":{\"quaternion\":[%.8f,%.8f,%.8f,%.8f],\"position\":[%.8f,%.8f,%.8f]},"
		"\"right_hand\":{\"quaternion\":[%.8f,%.8f,%.8f,%.8f],\"position\":[%.8f,%.8f,%.8f]}}",
		static_cast<unsigned long long>(info.FrameIndex),
		info.predictedDisplayTime,
		info.HeadPose_Pose_Orientation.x,
		info.HeadPose_Pose_Orientation.y,
		info.HeadPose_Pose_Orientation.z,
		info.HeadPose_Pose_Orientation.w,
		info.HeadPose_Pose_Position.x,
		info.HeadPose_Pose_Position.y,
		info.HeadPose_Pose_Position.z,
		info.LeftHand_Pose_Orientation.x,
		info.LeftHand_Pose_Orientation.y,
		info.LeftHand_Pose_Orientation.z,
		info.LeftHand_Pose_Orientation.w,
		info.LeftHand_Pose_Position.x,
		info.LeftHand_Pose_Position.y,
		info.LeftHand_Pose_Position.z,
		info.RightHand_Pose_Orientation.x,
		info.RightHand_Pose_Orientation.y,
		info.RightHand_Pose_Orientation.z,
		info.RightHand_Pose_Orientation.w,
		info.RightHand_Pose_Position.x,
		info.RightHand_Pose_Position.y,
		info.RightHand_Pose_Position.z);
}
