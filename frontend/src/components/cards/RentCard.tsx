import { RentCardModel } from "@/types/cards";

type Props = {
  card: RentCardModel;
  count?: number;
};

const getRentColor = (name?: string) => {
  switch (name) {
    case "BROWN":
      return "from-amber-800 to-amber-600";
    case "GREEN":
      return "from-green-600 to-green-400";
    case "PINK":
      return "from-pink-500 to-pink-300";
    case "BLUE":
      return "from-blue-600 to-blue-400";
    case "YELLOW":
      return "from-yellow-500 to-yellow-300";
    case "RED":
      return "from-red-600 to-red-400";
    case "ORANGE":
      return "from-orange-500 to-orange-300";
    case "PURPLE":
      return "from-purple-600 to-purple-400";
    default:
      return "from-gray-600 to-gray-400";
  }
};

const getRentBorderColor = (name?: string) => {
  switch (name) {
    case "BROWN":
      return "border-amber-400/40";
    case "GREEN":
      return "border-green-400/40";
    case "PINK":
      return "border-pink-400/40";
    case "BLUE":
      return "border-blue-400/40";
    case "YELLOW":
      return "border-yellow-400/40";
    case "RED":
      return "border-red-400/40";
    case "ORANGE":
      return "border-orange-400/40";
    case "PURPLE":
      return "border-purple-400/40";
    default:
      return "border-gray-400/40";
  }
};

export function RentCard({ card, count }: Props) {
  const bg = getRentColor(card.property_name);
  const border = getRentBorderColor(card.property_name);

  return (
    <div className={`bg-white text-gray-900 rounded-lg p-3 border ${border} w-[120px] h-[160px] flex flex-col justify-between relative`}>
      {/* Count badge in top right */}
      {count !== undefined && count > 1 && (
        <div className="absolute top-1 right-1 text-white text-[10px] font-bold rounded-full w-7 h-7 flex items-center justify-center border-2 border-white shadow-sm z-10 bg-blue-500">
          <span className="leading-none">{count}</span>
        </div>
      )}

      <div className="text-[10px] font-medium text-gray-500 uppercase tracking-wide py-1">Rent</div>

      <div className="text-center flex-1 flex flex-col items-center justify-center">
        <div className={`h-4 w-full rounded bg-gradient-to-r ${bg} mb-2`}></div>
        <div className="text-lg font-bold">{card.name || "Unknown"}</div>
      </div>

      <div></div>
    </div>
  );
}
