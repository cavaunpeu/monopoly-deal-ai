import { PropertyCardModel } from "@/types/cards";

type Props = {
  card: PropertyCardModel;
  count?: number; // Current count of this property type
  showProgress?: boolean; // Whether to show progress format (count/total) or just raw count
};

const getPropertyColor = (name?: string) => {
  switch (name) {
    case "BROWN":
      return "from-amber-800 to-amber-600";
    case "GREEN":
      return "from-green-600 to-green-400";
    case "PINK":
      return "from-pink-500 to-pink-300";
    default:
      return "from-gray-600 to-gray-400";
  }
};

const getBorderColor = (name?: string) => {
  switch (name) {
    case "BROWN":
      return "border-amber-400/40";
    case "GREEN":
      return "border-green-400/40";
    case "PINK":
      return "border-pink-400/40";
    default:
      return "border-gray-400/40";
  }
};

export function PropertyCard({ card, count, showProgress = false }: Props) {
  const bg = getPropertyColor(card.name);
  const border = getBorderColor(card.name);

  // Calculate needed count from rent progression length
  const neededCount = card.rent_progression?.length || 0;
  const currentCount = count || 0;
  const isComplete = currentCount >= neededCount;

  return (
    <div className={`bg-white text-gray-900 rounded-lg p-3 border ${border} w-[120px] h-[160px] flex flex-col justify-between relative`}>
      {/* Count badge in top right */}
      {count !== undefined && (
        <div className={`absolute top-1 right-1 text-white text-[10px] font-bold rounded-full w-7 h-7 flex items-center justify-center border-2 border-white shadow-sm z-10 ${
          showProgress && isComplete
            ? 'bg-green-500' // Green for completed sets (only in properties section)
            : 'bg-blue-500'  // Blue for in progress or raw count
        }`}>
          <span className="leading-none">
            {showProgress ? `${currentCount}/${neededCount}` : currentCount}
          </span>
        </div>
      )}

      <div className="text-[10px] font-medium text-gray-500 uppercase tracking-wide py-1">Property</div>

      <div className="text-center flex-1 flex flex-col items-center justify-center">
        <div className={`h-4 w-full rounded bg-gradient-to-r ${bg} mb-2`}></div>
        <div className="text-lg font-bold">{card.name || "Unknown"}</div>
        <div className="text-[10px] text-gray-600 mt-1">Rent {card.rent_progression?.join(" → ") || "?"}</div>
      </div>

      <div className="text-center">
        <div className="text-xs font-bold text-gray-700">${card.value || 0}</div>
      </div>
    </div>
  );
}
